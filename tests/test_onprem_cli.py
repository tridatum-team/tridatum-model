import contextlib
import hashlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from trid.onprem import __main__ as cli
from trid.onprem.config import load_config
from trid.onprem.store import Store

PW = 'correct horse 1'


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.tok = self.root / 'tokenizer.json'
        self.tok.write_text('{"secret-tokenizer-content": 1}', encoding='utf-8')
        self.cfg = self.root / 'onprem.json'
        self.write_config()

    def tearDown(self):
        self.tmp.cleanup()

    def write_config(self, **extra):
        data = {'hostname': 'x.example', 'data_dir': str(self.root), 'tokenizer_path': str(self.tok), 'port': 0}
        data.update(extra)
        self.cfg.write_text(json.dumps(data), encoding='utf-8')

    def run_cli(self, argv, stdin=''):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                mock.patch('sys.stdin', io.StringIO(stdin)):
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def store(self):
        return Store(self.root / 'onprem.db')

    def test_create_admin_via_password_stdin(self):
        code, out, err = self.run_cli(['create-admin', '--config', str(self.cfg), '--username', 'Root',
                                       '--password-stdin'], PW + '\n')
        self.assertEqual(code, 0, err)
        store = self.store()
        user = store.get_user('root')
        self.assertEqual((user.role, user.must_change), ('admin', False))
        events = store.query_events(event='admin_create_user')
        self.assertEqual(len(events), 1)
        self.assertIsNone(events[0]['username'])
        self.assertEqual(events[0]['detail'], {'target': 'root', 'action': 'create'})
        self.assertNotIn(PW, out + err)
        store.close()

    def test_create_admin_refuses_existing_user(self):
        argv = ['create-admin', '--config', str(self.cfg), '--username', 'root', '--password-stdin']
        self.assertEqual(self.run_cli(argv, PW + '\n')[0], 0)
        code, out, err = self.run_cli(argv, PW + '\n')
        self.assertNotEqual(code, 0)
        self.assertIn('이미', err)

    def test_create_admin_getpass_must_match(self):
        argv = ['create-admin', '--config', str(self.cfg), '--username', 'root']
        with mock.patch('getpass.getpass', side_effect=[PW, PW + 'x']):
            code, _, err = self.run_cli(argv)
        self.assertNotEqual(code, 0)
        store = self.store()
        self.assertIsNone(store.get_user('root'))
        store.close()
        with mock.patch('getpass.getpass', side_effect=[PW, PW]):
            self.assertEqual(self.run_cli(argv)[0], 0)

    def test_serve_refuses_tokenizer_with_wrong_sha256(self):
        for bind in ([], ['--bind', '127.0.0.1']):
            with self.subTest(bind=bind), mock.patch.object(cli, 'make_server') as make:
                code, out, err = self.run_cli(['serve', '--config', str(self.cfg), *bind])
                self.assertNotEqual(code, 0)
                self.assertIn(cli.TOKENIZER_SHA256, err)
                self.assertNotIn('secret-tokenizer-content', out + err)
                make.assert_not_called()

    def test_serve_insecure_dev_warns_on_mismatch_and_binds_loopback(self):
        for bind in ([], ['--bind', '127.0.0.1']):
            with self.subTest(bind=bind):
                server = mock.Mock(server_address=("127.0.0.1", 1))
                server.serve_forever.side_effect = KeyboardInterrupt
                with mock.patch.object(cli, 'make_server', return_value=server) as make, \
                        mock.patch.object(cli, 'load_tokenizer', return_value=object()):
                    code, out, err = self.run_cli(['serve', '--config', str(self.cfg), '--insecure-dev', *bind])
                self.assertEqual(code, 0, err)
                self.assertIn(cli.TOKENIZER_SHA256, err)
                self.assertEqual(make.call_args.kwargs['host'], '127.0.0.1')
                self.assertTrue(make.call_args.kwargs['insecure_dev'])
                server.server_close.assert_called_once()
                self.assertNotIn('Traceback', err)

    def test_serve_insecure_dev_rejects_incompatible_bind_before_startup(self):
        for host in ('0.0.0.0', '192.0.2.1', 'localhost', '::1', ''):
            with self.subTest(host=host), \
                    mock.patch.object(cli, 'load_config') as config, \
                    mock.patch.object(cli, 'make_server') as make:
                code, _, err = self.run_cli([
                    'serve', '--config', str(self.cfg), '--insecure-dev', '--bind', host,
                ])
                self.assertNotEqual(code, 0)
                self.assertIn('--bind 127.0.0.1', err)
                config.assert_not_called()
                make.assert_not_called()

    def test_serve_production_bind_does_not_enable_insecure_dev(self):
        for bind, expected_host in (([], '0.0.0.0'), (['--bind', '127.0.0.1'], '127.0.0.1')):
            with self.subTest(bind=bind):
                server = mock.Mock(server_address=(expected_host, 1))
                server.serve_forever.side_effect = KeyboardInterrupt
                with mock.patch.object(cli, 'make_server', return_value=server) as make, \
                        mock.patch.object(cli, 'load_tokenizer', return_value=object()), \
                        mock.patch.object(cli, 'TOKENIZER_SHA256', hashlib.sha256(self.tok.read_bytes()).hexdigest()):
                    code, _, err = self.run_cli(['serve', '--config', str(self.cfg), *bind])
                self.assertEqual(code, 0, err)
                self.assertEqual(make.call_args.kwargs['host'], expected_host)
                self.assertFalse(make.call_args.kwargs['insecure_dev'])
                server.server_close.assert_called_once()

    def test_purge_respects_retention_days(self):
        self.write_config(retention_days=10)
        now = time.time()
        store = self.store()
        store.log('clear', None, None, 'ok')
        store._db.execute('UPDATE events SET ts=?', (now - 11 * 86400,))
        store.log('clear', None, None, 'ok')
        store.close()
        code, out, err = self.run_cli(['purge', '--config', str(self.cfg)])
        self.assertEqual(code, 0, err)
        store = self.store()
        names = [(e['event'], e['detail'].get('count')) for e in store.query_events()]
        self.assertIn(('retention_purge', 1), names)
        self.assertEqual(len([n for n in names if n[0] == 'clear']), 1)
        store.close()

    def test_load_config_rejects_unknown_keys(self):
        self.write_config(bogus=1)
        with self.assertRaises(ValueError):
            load_config(self.cfg)
        code, out, err = self.run_cli(['purge', '--config', str(self.cfg)])
        self.assertNotEqual(code, 0)
        self.assertNotIn('Traceback', err)

    def test_load_config_rejects_non_positive_or_bool_retention(self):
        for bad in (0, -1, True):
            self.write_config(retention_days=bad)
            with self.assertRaises(ValueError):
                load_config(self.cfg)
        self.write_config(retention_days=1)
        self.assertEqual(load_config(self.cfg).retention_days, 1)

    def test_offline_env_is_forced(self):
        import importlib
        with mock.patch.dict('os.environ', {'HF_HUB_OFFLINE': '0', 'TOKENIZERS_PARALLELISM': 'true'}):
            importlib.reload(cli)
            import os
            self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')
            self.assertEqual(os.environ['TOKENIZERS_PARALLELISM'], 'false')

    def test_example_config_loads(self):
        path = Path(__file__).resolve().parent.parent / 'deploy' / 'onprem' / 'onprem.example.json'
        self.assertEqual(load_config(path).hostname, 'trid.example.internal')


if __name__ == '__main__':
    unittest.main()
