"""Local browser route editor. Run with python -m src.tools.route_editor."""
import argparse
import copy
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets
import tempfile
import webbrowser

import yaml

from src.data.population import KINDS, validate_population
from src.data.route_csv import read_tracks, validate_tracks, write_tracks, speed_warnings
from src.data.scenarios import project_path, write_inputs
from src.data.generate_handover_batch import validate


class EditorProject:
    def __init__(self, config_path, training_path, output, output_config=None):
        self.config = yaml.safe_load(project_path(config_path).read_text())
        validate(self.config)
        self.training = yaml.safe_load(project_path(training_path or self.config.get('training_template', 'configs/pipeline.yaml')).read_text())
        self.output = project_path(output)
        self.output_config = project_path(output_config) if output_config else None
        env = self.config['environment']
        self.profiles = self.training['ue_attributes'] if training_path else env.get('ue_attributes', self.training['ue_attributes'])
        self.tracks = {}
        with tempfile.TemporaryDirectory() as tmp:
            for scenario in env['scenarios']:
                path = Path(tmp) / scenario['name']
                write_inputs(env, scenario, path)
                self.tracks[scenario['name']] = read_tracks(path, env)

    def state(self):
        env = self.config['environment']
        from src.data.population import population_offsets
        population = env.get('population', {'min_ues': 1, 'max_ues': 12,
                         'counts': {k: len(v) for k, v in population_offsets(env).items()}})
        return dict(sites=env['sites'], duration=env['duration_s'], profiles=self.profiles,
                    bounds=self.training['model']['bounds'], population=population,
                    scenarios=[{'name': s['name'], 'partition': s['partition']} for s in env['scenarios']],
                    tracks=self.tracks)

    def save(self, payload):
        cfg = copy.deepcopy(self.config)
        env = cfg['environment']
        population = copy.deepcopy(self.state()['population'])
        population['counts'] = payload['counts']
        validate_population(population)
        env['population'] = population
        for key in ('car_offsets', 'ue_offsets'):
            env.pop(key, None)
        env['ue_attributes'] = self.profiles
        if set(payload['tracks']) != {s['name'] for s in env['scenarios']}:
            raise ValueError('Scenario set differs from the configuration')
        validated = {name: validate_tracks(tracks, env) for name, tracks in payload['tracks'].items()}
        messages = [f'{name}: {w}' for name, tracks in validated.items() for w in speed_warnings(tracks, self.profiles)]
        # A revision is immutable. Saving edits cannot change an existing simulation plan.
        revision = datetime.now().strftime('%Y%m%d_%H%M%S_') + secrets.token_hex(3)
        root = self.output / 'revisions' / revision
        root.mkdir(parents=True, exist_ok=False)
        for scenario in env['scenarios']:
            name = scenario['name']
            path = root / 'inputs' / name
            write_tracks(path, validated[name], env)
            for key in ('car_route', 'walker_route', 'static_position', 'route_csv_dir'):
                scenario.pop(key, None)
        env['route_csv_root'] = str(root / 'inputs')
        template = root / 'training_template.yaml'
        template.write_text(yaml.safe_dump(self.training, sort_keys=False))
        cfg['training_template'] = str(template)
        cfg['batch_dir'] = str(project_path('data/processed') / self.output.name / revision)
        text = yaml.safe_dump(cfg, sort_keys=False)
        (root / 'simulation.yaml').write_text(text)
        latest = self.output_config or self.output / 'simulation.yaml'
        latest.parent.mkdir(parents=True, exist_ok=True)
        tmp = latest.with_suffix('.yaml.tmp')
        tmp.write_text(text)
        tmp.replace(latest)
        self.config = cfg
        self.tracks = validated
        return dict(config=str(latest), revision_config=str(root / 'simulation.yaml'), latest_config=str(latest),
                    batch_dir=cfg['batch_dir'], warnings=messages)


def make_handler(project, token):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, content, kind='application/json'):
            body = content.encode('utf-8') if isinstance(content, str) else json.dumps(content, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', kind + '; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/':
                html = Path(__file__).with_name('route_editor.html').read_text().replace('__TOKEN__', token)
                self.reply(200, html, 'text/html')
            elif self.path == '/api/state':
                self.reply(200, project.state())
            else:
                self.reply(404, {'error': 'Not found'})

        def do_POST(self):
            if self.path != '/api/save' or self.headers.get('X-Editor-Token') != token:
                self.reply(403, {'error': 'Invalid request'})
                return
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 10_000_000:
                    raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(size))
                self.reply(200, project.save(payload))
            except (ValueError, TypeError, KeyError, OSError) as error:
                self.reply(400, {'error': str(error)})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/training_dataset_v2.yaml')
    parser.add_argument('--training-config', help='Profiles and initial-position bounds; defaults to pipeline.yaml or saved template')
    parser.add_argument('--output-dir', default='data/scenarios/manual_routes')
    parser.add_argument('--output-config', default='configs/routes_editor.yaml', help='YAML pointing to the saved CSV folder')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    project = EditorProject(args.config, args.training_config, args.output_dir, args.output_config)
    with HTTPServer(('127.0.0.1', args.port), make_handler(project, secrets.token_urlsafe(32))) as server:
        url = f'http://127.0.0.1:{server.server_port}'
        print(f'Route editor: {url}\nOutput: {project.output}\nCtrl+C to stop.', flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
