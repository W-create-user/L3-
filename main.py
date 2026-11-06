"""Run all four agents against an installed SMPL source checkout."""
import argparse
import json
from runtime.cluster import AgentCluster
from runtime.environment_adapter import EnvironmentAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal', default='执行一次控制任务')
    parser.add_argument('--smpl-source', help='Directory containing smpl/envs and smpl/configdata')
    parser.add_argument('--seed', type=int, default=1101)
    args = parser.parse_args()
    with EnvironmentAdapter(source_dir=args.smpl_source) as adapter:
        result = AgentCluster(adapter).run(args.goal, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
