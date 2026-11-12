"""Fill hashes from supplied files and write a DRAFT for human review, never approval."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from intake_dataset import digest, read_json


def prepare(spec, root, config_path, dictionary_path):
    result=deepcopy(spec)
    result['status']='draft'
    result['collection_config_sha256']=digest(config_path)
    result['variable_dictionary_sha256']=digest(dictionary_path)
    root=Path(root).resolve()
    for episode in result['episodes']:
        for artifacts in episode['artifacts'].values():
            for artifact in artifacts:
                path=(root/artifact['path']).resolve()
                if not path.is_relative_to(root):raise ValueError('artifact path outside dataset root')
                artifact['sha256']=digest(path)
    schema=read_json(Path(__file__).resolve().parents[1]/'data_manifest.schema.json')
    errors=list(Draft202012Validator(schema,format_checker=FormatChecker()).iter_errors(result))
    if errors:raise ValueError(f'manifest structure invalid: {errors[0].message}')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec',type=Path)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--dictionary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=prepare(read_json(args.spec),args.root,args.config,args.dictionary)
        with args.output.open('x',encoding='utf-8') as stream:
            stream.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    except (OSError,UnicodeError,ValueError,KeyError,TypeError) as exc:
        parser.exit(1,f'Cannot prepare draft: {type(exc).__name__}: {exc}\n')
    print('Draft created. Human source/split/permission review is still required; it is not an accepted dataset.')


if __name__=='__main__':
    main()
