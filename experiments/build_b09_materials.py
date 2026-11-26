"""Generate source-bound topology, methods and dependency/license evidence.

Outputs are research writing inputs, not field calibration or legal approval.
No plant is instantiated and no external data/weights are downloaded.
"""
import argparse
import csv
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import platform
import subprocess
from .run_b08_development import ROOT, write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mpctools-source',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    source=json.loads((ROOT/'vendor/smpl/SOURCE_VERSION.json').read_text())
    vendor_status={path:hashlib.sha256((ROOT/'vendor/smpl'/path).read_bytes()).hexdigest()==expected
        for path,expected in source['files'].items()}
    dictionary=ROOT/'docs/data_dictionary.csv'
    if not dictionary.exists():raise FileNotFoundError('B companion source dictionary required')
    with dictionary.open() as f:rows=list(csv.DictReader(f))
    provenance=json.loads((ROOT/'configs/environments/dictionary_provenance.json').read_text())
    if hashlib.sha256(dictionary.read_bytes()).hexdigest()!=provenance['dictionary_sha256']:raise ValueError('Dictionary changed')
    license_dir=a.output/'license_evidence';license_dir.mkdir()
    for label,file in [('SMPL_MIT',ROOT/'vendor/smpl/LICENSE.rst'),('MPCTools_COPYING',a.mpctools_source/'COPYING.txt')]:
        if file.exists():(license_dir/(label+'.txt')).write_bytes(file.read_bytes())
    deps=[]
    for dist in sorted(md.distributions(),key=lambda d:d.metadata.get('Name','').lower()):
        licenses=[str(f) for f in (dist.files or []) if 'license' in str(f).lower() or 'copying' in str(f).lower()]
        deps.append({'name':dist.metadata.get('Name'),'version':dist.version,
            'license_expression':dist.metadata.get('License-Expression'),
            'declared_license':dist.metadata.get('License'),
            'license_classifiers':[s for s in dist.metadata.get_all('Classifier',[]) if s.startswith('License')],
            'installed_license_paths':licenses,'redistribution_review':'pending package-specific review'})
    write(a.output/'dependencies_and_licenses.json',{'python':platform.python_version(),'packages':deps,
        'source_licenses':{'SMPL':'MIT: retained copyright and text',
            'MPCTools':'Source COPYING is GNU GPL v3; distribution/composition review pending'},
        'not_legal_clearance':True})
    anchors={
        'buffer_geometry':'D_bf = 5', 'buffer_inflow':'F_in_bf = u[3]',
        'buffer_outflow':'F_out_bf = u[7]', 'buffer_state':'h = x[17]',
        'buffer_equation':'dhdt = 1 / Ac_bf * (F_in_bf - F_out_bf)'}
    lines=(ROOT/'vendor/smpl/smpl/envs/helpers/mab_helpers.py').read_text().splitlines()
    locations={key:{'path':'vendor/smpl/smpl/envs/helpers/mab_helpers.py',
        'line':next(i for i,s in enumerate(lines,1) if text in s),'expression':text} for key,text in anchors.items()}
    report={'date':'2026-11-26','source_upstream':source['upstream'],'source_commit':source['commit'],
        'matches_upstream_source_manifest':all(vendor_status.values()),'vendor_file_checks':vendor_status,
        'source_manifest_mismatches':{path:{'declared':source['files'][path],'actual':hashlib.sha256((ROOT/'vendor/smpl'/path).read_bytes()).hexdigest()} for path,ok in vendor_status.items() if not ok},
        'baseline_vendor_unchanged':not bool(subprocess.check_output(['git','diff','4a09dcd','--','vendor/smpl'],cwd=ROOT,text=True)),
        'dictionary_rows':len(rows),'dictionary_sha256':provenance['dictionary_sha256'],
        'model':'MAbEnvGym in mabenv.py; integrated/old variants are not the selected runtime',
        'model_coordinates':{'observations':1970,'actions':9,'sampling_period_min':60,
            'observation_mapping':'physical = model_observation * xscale',
            'action_mapping':'physical = model_action * uscale',
            'source_blocks':'17 upstream states + 2 buffer states + 1951 capture/discretized/integral states'},
        'topology':'source bioreactor/separation recycle -> buffer -> modeled capture column',
        'source_locations':locations,'provenance':provenance,
        'excluded':['warehousing','logistics','formulation and filling validation','full purification train validation',
                    'field calibrated digital twin','real energy consumption without an audited energy model'],
        'unknown':['q1/q2 state units','last accumulation state physical mass conversion',
                   'reliable physical hazard thresholds','field sensor equivalence','full production quality acceptance'],
        'B09_status':'source-bound writing inputs generated; D08 final statistics and team/mentor review pending',
        'B10_status':'image and clean independent environment acceptance not yet done',
        'dependency_review':'SMPL MIT, MPCTools source GPLv3; do not assume all dependencies permissive',
        'source_code_changes':'B adapter/controller/gateway only; vendor unchanged relative to A main 4a09dcd. Historical SOURCE_VERSION utils.py hash mismatch retained for review',
        'data_rights':'B05 simulation/synthetic only; no field data or independent field imagery claimed'}
    write(a.output/'model_boundaries.json',report)
    (a.output/'process_topology.mmd').write_text('''flowchart LR
  Feed["Model feed actions"] --> Up["Source upstream reactor and separation/recycle"]
  Up -->|"physical inflow u3 L/min"| Buffer["Buffer: h=x17 dm, concentration=x18 mg/L"]
  Buffer -->|"physical drain u7 L/min"| Capture["Modeled capture column; action u8 switches capture"]
  Capture --> Output["Model output/integral; mass interpretation pending"]
''')
    (a.output/'methods_draft.md').write_text('''# B09 方法章节输入（待联合复核）

研究采用固定版本SMPL科研仿真，不称现场校准数字孪生。选择mabenv.py中的MAbEnvGym，1970维模型观测、9维动作、采样60仿真分钟；物理坐标由xscale/uscale转换。当前vendor与A主分支4a09dcd一致，但utils.py与历史SOURCE_VERSION声明哈希不一致，未冒称完全等同上游固定提交。实际哈希、范围与行号见model_boundaries.json，独立模型分支和旧版本不能混用。

控制基线为D24公开比例控制，候选为同P控制加可信启动质量衡算估计；另以B冻结PI作开发对照。新候选借鉴D25 BufferObserver思路，B实现实际动作回执绑定、版本核对和事务提交。只在已声明可信、故障前的初始标定成立时使用；随后只接收投递液位及已确认执行流量，不读取扰动标签、退出时刻或评分真值。未知初始偏置、未测流量扰动和模型偏差不能由本试验排除。

开发对照沿用公开26001—26030条件，三工况各10组，每组四种控制/编排共120次；这不是新保留组，也不是正式盲测。工况为正常、前三步测量加0.10 dm、正负0.15—0.30 dm初态偏移；初态偏移不等于持续过程扰动。目标14.95—15.05 dm，连续3个有效post-step、至多10步，90秒墙钟预算，失败及未启动全部保留。同P候选直接执行和A/C单资源执行用于控制归因，不能推导分布式调度收益。

状态、请求/实际动作、模型裁剪、版本、失败原因和耗时逐条归档。成功率同时报告已启动与计划分母；共同成功配对用仿真分钟均值之比计算周期缩短，另报所有计划配对失败按600分钟惩罚的敏感性结果。分工况配对重采样只作开发描述，非独立现场泛化证据，20%目标不据开发结果宣布验收通过。

模型门控为D已公开的定义域围栏，不能将Box范围称为物理安全阈值。D学习模型和C多资源契约未经联合冻结，未在本开发对照中冒充启用。B05数值、文字及图像来自仿真与数值派生，传感器标签不是工艺健康真值。

第三方许可证据随目录保存。SMPL为MIT，MPCTools源码包含GPLv3文本；分发及组合许可仍需逐包复核。没有真实发表、授权或独立复现凭证时不宣称已取得。D08最终统计、导师边界复核和B10镜像验收仍待后续完成。
''')
    print(json.dumps({'source_unchanged':report['baseline_vendor_unchanged'],'dictionary_rows':len(rows),'packages':len(deps)}))


if __name__=='__main__':main()
