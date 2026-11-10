"""Terminal progress resumes from durable records and closes on interruption."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tracegen.progress import Progress

class Terminal(io.StringIO):
    def isatty(self):
        return True

class ProgressTests(unittest.TestCase):
    def test_resume_selected_total_and_interruption_display(self):
        stream = Terminal()
        with tempfile.TemporaryDirectory() as directory, patch('sys.stderr', stream):
            progress = Progress(Path(directory), {'run_id': 'run'},
                                [{'problem_id': 'a'}] * 3 + [{'problem_id': 'b'}] * 3,
                                [{'problem_id': 'a', 'status': 'ok'},
                                 {'problem_id': 'b', 'status': 'ok'}], {'a'}, 'session')
            progress.publish(state='running')
            self.assertIn('1/3', stream.getvalue())
            progress.saved({'problem_id': 'a', 'status': 'ok', 'error': None})
            progress.publish(state='interrupted', log=True)
            self.assertIn('2/3', stream.getvalue())
            self.assertIn('interrupted', stream.getvalue())
            self.assertTrue(stream.getvalue().endswith('\n'))
