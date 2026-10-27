"""s02_preprocess/token_mapping.py is Yanping's module, copied unchanged from origin/yanping, and the cache builder uses it.

The behaviour ids of every cache come from this file; a silent edit would make the next cache differ from the existing ones.
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / 's02_preprocess' / 'token_mapping.py'
# sha256 of CT_GAT/preprocessing/token_mapping.py as stored in git (line feeds), identical on every branch that has it
SHA256 = '8da1001501cf88228f6bf34b2cfe0829f18cd71730cc4bf43631eb6930be56e3'


def test_the_vendored_file_is_the_unchanged_original():
    """A checkout with CRLF line endings (core.autocrlf) is read as the LF text git stores."""
    text = VENDORED.read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(text).hexdigest() == SHA256


def test_build_cache_uses_that_module_and_needs_no_environment_variable():
    from s02_preprocess import token_mapping
    from s02_preprocess.cache import build_cache
    assert build_cache.behavior_rules() is token_mapping
    assert sorted(token_mapping.BEHAVIOR_MAP, key=token_mapping.BEHAVIOR_MAP.get)[:2] == ['NONE', 'CRYPTO_OPCODE']
    for name in ('categorize_behavior', '_match_api_in_operands', 'strip_opcode_prefixes'):
        assert callable(getattr(token_mapping, name))


def test_build_cache_imports_with_no_environment_at_all(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GT_')}
    env.update(GT_DATASETS=str(tmp_path / 'none.json'), PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run([sys.executable, '-c', 'from s02_preprocess.cache import build_cache; '
                        'print(build_cache.behavior_rules().BEHAVIOR_MAP["NONE"])'],
                       cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0 and r.stdout.strip() == '0', r.stderr
