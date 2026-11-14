"""RGB camera rendering shared by the MCP and Nemotron interfaces."""

import json
import pathlib
import subprocess


CAMERAS = (('orbit', 'Orbit camera'), ('blade', 'Blade camera'))
RUNNER = pathlib.Path(__file__).with_name('render-cameras.mjs')


class Cameras:
    def __init__(self, env):
        self.env = env
        self.process = None
        self.frame = None
        self.pictures = None

    def __enter__(self):
        try:
            self.process = subprocess.Popen(
                ['node', str(RUNNER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, encoding='utf-8',
            )
        except FileNotFoundError as error:
            raise RuntimeError('RGB cameras require Node.js. See the setup in README.md.') from error
        try:
            self.request(self.env.replay())
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        finally:
            self.process.stdout.close()

    def request(self, payload):
        try:
            self.process.stdin.write(json.dumps(payload) + '\n')
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        except BrokenPipeError:
            line = ''
        if not line:
            raise RuntimeError('RGB renderer exited. Run npm ci and npx playwright install chromium.')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(f'RGB renderer: {result["error"]}')
        return result

    def images(self):
        frame = self.env.log[-1]
        if frame is not self.frame:
            self.pictures = self.request({key: frame[key] for key in ('arm', 'cam', 'len')})
            self.frame = frame
        return self.pictures
