"""Check repository Markdown relative file links, with no network requests."""
import re
from pathlib import Path
from urllib.parse import unquote


def main():
    root=Path(__file__).resolve().parents[1]
    documents=[root/'README.md',*sorted((root/'docs').glob('*.md'))]
    missing=[]
    for document in documents:
        for target in re.findall(r'\]\(([^)]+)\)',document.read_text(encoding='utf-8')):
            target=unquote(target.split('#',1)[0])
            if not target or re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*:',target):
                continue
            if not (document.parent/target).exists():
                missing.append(f'{document.relative_to(root)}: {target}')
    if missing:
        raise RuntimeError('Missing documentation targets:\n'+'\n'.join(missing))
    print(f'Checked relative file links in {len(documents)} documents')


if __name__=='__main__':
    main()
