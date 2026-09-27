"""Project paths from a local .env file, with process environment overrides."""
from dataclasses import dataclass
from pathlib import Path
import os
import shlex

@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    raw_data_dir: Path
    prepared_data_dir: Path
    models_dir: Path
    reports_dir: Path


def load_paths(project_root=None, env_file=None):
    """Relative paths resolve against the project root, not notebook cwd.

    .env accepts KEY=value, quoted values and comments. Variable interpolation
    and multiline values are deliberately unsupported. No environment is mutated.
    Shell environment values override file values. Directories are not created.
    """
    root = Path(project_root).expanduser().resolve() if project_root else Path(__file__).resolve().parents[1]
    file = Path(env_file).expanduser() if env_file else root / '.env'
    if not file.is_absolute():
        file = root / file
    if not file.exists():
        raise FileNotFoundError(f'Copy .env.example to {file}, then configure your paths.')
    values = {}
    allowed = {'RAW_DATA_DIR','PREPARED_DATA_DIR','MODELS_DIR','REPORTS_DIR'}
    for number,line in enumerate(file.read_text().splitlines(),1):
        line=line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            raise ValueError(f'{file.name}:{number}: expected KEY=value')
        key,value=line.split('=',1);key=key.strip()
        if key not in allowed:
            raise ValueError(f'{file.name}:{number}: unsupported path key {key}')
        parts=shlex.split(value,comments=True,posix=True)
        if len(parts)!=1 or '$' in parts[0]:
            raise ValueError(f'{file.name}:{number}: use one literal path; quote spaces')
        values[key]=parts[0]
    def resolve(key):
        value=os.environ.get(key,values.get(key,''))
        if not value:
            raise ValueError(f'Missing path setting: {key}')
        p=Path(value).expanduser()
        return (root/p).resolve() if not p.is_absolute() else p.resolve()
    return ProjectPaths(root,*(resolve(k) for k in ('RAW_DATA_DIR','PREPARED_DATA_DIR','MODELS_DIR','REPORTS_DIR')))
