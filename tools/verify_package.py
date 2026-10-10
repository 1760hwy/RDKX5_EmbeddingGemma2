"""Local source-package verification; no model inference or network access."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
errors = []
files = [p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts]
for path in files:
    if path.suffix == '.py':
        try:
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        except SyntaxError as exc:
            errors.append(f'Syntax: {path.relative_to(ROOT)}: {exc}')
    if path.suffix == '.md':
        for link in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if '://' in link or link.startswith('#'):
                continue
            target = (path.parent / link.split('#')[0]).resolve()
            if not target.exists():
                errors.append(f'Broken link: {path.relative_to(ROOT)} -> {link}')
    if path.suffix in ('.md', '.py', '.cpp', '.sh', '.json', '.html', '.css', '.js', '.txt'):
        value = path.read_text(encoding='utf-8-sig')
        patterns = (r'hf_[A-Za-z0-9]{20,}', r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',
                    r'(?i)\b(?:password|passwd)\s*[:=]\s*["\x27][^"\x27]+["\x27]',
                    r'(?i)(?:plink|pscp)[^\n]+\s-pw\s+', r'192\.168\.\d+\.\d+')
        # This verifier necessarily contains the patterns, not credentials.
        if path.name != 'verify_package.py':
            for pattern in patterns:
                if re.search(pattern, value):
                    errors.append('Sensitive pattern needs review: ' + str(path.relative_to(ROOT)))
    if path.suffix in ('.bin', '.onnx', '.hbm', '.hbir', '.so', '.safetensors', '.whl'):
        errors.append('Excluded binary present: ' + str(path.relative_to(ROOT)))

manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
for record in manifest['source_records']:
    path = ROOT / record['export']
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record['export_sha256']:
        errors.append('Export SHA mismatch: ' + record['export'])
result = {'passed': not errors, 'errors': errors, 'files_checked': len(files),
          'python_syntax_checked': sum(p.suffix == '.py' for p in files),
          'export_sha_records_checked': len(manifest['source_records']),
          'network_or_board_tests_performed': False,
          'licensing_approved': (ROOT / 'LICENSE').read_text(encoding='utf-8').startswith('MIT License'), 'clean_environment_reproduction_tested': False}
(ROOT / 'PACKAGE_CHECKS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(int(bool(errors)))
