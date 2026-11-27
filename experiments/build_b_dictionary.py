"""Source-derived variable dictionary; model bounds are never called safety limits."""
import csv
import json
from pathlib import Path
import hashlib
from l3_system.execution import EnvironmentAdapter, MAbAdapter
from l3_system.planning.goals import validate_registry

ROOT=Path(__file__).resolve().parents[1]

def main():
    output=ROOT/'docs/data_dictionary.csv';rows=[]
    def source(path,needle):
        lines=(ROOT/path).read_text().splitlines()
        return path,next(i for i,line in enumerate(lines,1) if needle in line)
    helper='vendor/smpl/smpl/envs/helpers/mab_helpers.py'
    formula=source(helper,'x_reshape = cs.reshape')
    def add(adapter,kind,index,name,unit,scale,offset,path,line,status='source_declared',note=''):
        space=adapter.observation_space if kind=='observation' else adapter.action_space
        rows.append(dict(backend=adapter.metadata['environment_id'],kind=kind,index=index,name=name,
                         returned_coordinate=adapter.metadata['observation_coordinate' if kind=='observation' else 'action_coordinate'],
                         physical_unit=unit,unit_status=status,physical_equals_model_times_scale_plus_offset=(status!='unconfirmed'),
                         scale=float(scale),offset=float(offset),model_low=float(space.low[index]),model_high=float(space.high[index]),
                         reliable_process_range='unconfirmed',bound_scope='model_Box_not_safety_or_reliability_limit',
                         observable_in_simulation=True,field_sensor_equivalence='unconfirmed',source_path=path,source_line=line,notes=note))
    with EnvironmentAdapter() as a:
        fields=json.loads((ROOT/'configs/goals/sources/observation_schema_verified.json').read_text())['fields']
        for f in fields:
            path,line=source('vendor/smpl/smpl/envs/atropineenv.py','observations = []')
            add(a,'observation',f['index'],f['name'],f['unit'],1,0,path,line,
                'unconfirmed' if 'unspecified' in f['unit'] else ('source_inferred' if 'inferred' in f['unit'] else 'source_declared'),
                'o[0] is preincrement index, not minutes; estimator coordinates are not physical sensors')
        for i,name in enumerate(['USS_1_delta','USS_2_delta','USS_3_delta','USS_4_delta']):
            path,line=source('vendor/smpl/smpl/envs/models/atropine_process.py',f'u[{i}] / 1000.0')
            add(a,'action',i,name,'mL/min',1,a.action_coordinates.offset[i],path,line,'source_inferred',
                'Inferred from conversion to L/min; upstream plot labels conflict (uL/min). Requires field calibration; no industrial-unit certification.')
    with MAbAdapter() as a:
        names=['Xv1','Xt1','GLC1','GLN1','LAC1','AMM1','mAb1','V1','Xv2','Xt2','GLC2','GLN2','LAC2','AMM2','mAb2','V2','T','buffer_height','buffer_mAb_concentration']
        units=['cell/L','cell/L','mM','mM','mM','mM','mg/L','L','cell/L','cell/L','mM','mM','mM','mM','mg/L','L','degC','dm','mg/L']
        needles=['Xv1 = x[0]','Xt1 = x[1]','GLC1 = x[2]','GLN1 = x[3]','LAC1 = x[4]','AMM1 = x[5]','mAb1 = x[6]','V1 = x[7]',
                 'Xv2 = x[8]','Xt2 = x[9]','GLC2 = x[10]','GLN2 = x[11]','LAC2 = x[12]','AMM2 = x[13]','mAb2 = x[14]','V2 = x[15]','T = x[16]','h = x[17]','c_bf = x[18]']
        for i,(name,unit,needle) in enumerate(zip(names,units,needles)):
            path,line=source(helper,needle);add(a,'observation',i,name,unit,a._xscale[i],0,path,line,
                note='Unit follows model comment only; model-wide chemical-unit consistency and field measurability not established')
        # CasADi reshape 13x150 column-major then transpose: node-major returned vector.
        for node in range(150):
            for compartment in range(13):
                index=19+13*node+compartment
                name=(f'capture_z{node}_mobile' if compartment==0 else f'capture_z{node}_particle_r{compartment-1}' if compartment<=10 else f'capture_z{node}_q{compartment-10}')
                add(a,'observation',index,name,'mg/L' if compartment<=10 else 'unknown',a._xscale[index],0,*formula,
                    status='source_declared' if compartment<=10 else 'unconfirmed',
                    note='Internal discretized state, not independently measured sensor; q concentration unit not explicitly confirmed')
        path,line=source('vendor/smpl/smpl/envs/mabenv.py','xkp1[-1] +=')
        add(a,'observation',1969,'capture_accumulation_integral','unknown',1,0,path,line,'unconfirmed',
            'Update integrates concentration difference over time; do not call this physical captured mass without a verified flow conversion')
        actionnames=['F_in','F_1','F_r','F_2','GLC_in','GLN_in','Tc','buffer_outlet','capture_switch']
        actionunits=['L/min','L/min','L/min','L/min','mM','mM','degC','L/min','dimensionless']
        needles=['F_in = u[0]','F_1 = u[1]','F_r = u[2]','F_2 = u[3]','GLC_in = u[4]','GLN_in = u[5]','Tc = u[6]','F_out_bf = u[7]']
        for i,(name,unit) in enumerate(zip(actionnames,actionunits)):
            path,line=source(helper,needles[i]) if i<8 else source('vendor/smpl/smpl/envs/mabenv.py','if uk[-1] >= self.switch_threshold:')
            add(a,'action',i,name,unit,a._uscale[i],0,path,line,note='physical=model*uscale; normalize=False is still scaled')
        variables={'buffer_height':{'source':'observation','index':17,'unit':'dm','coordinate':'physical_projection',
                                    'bounds':{'lower':float(a.observation_space.low[17]*a._xscale[17]),'upper':float(a.observation_space.high[17]*a._xscale[17])}}}
        for i,(name,unit) in enumerate(zip(actionnames,actionunits)):
            low=a.action_coordinates.to_physical(a.action_space.low);high=a.action_coordinates.to_physical(a.action_space.high)
            variables[name]={'source':'action','index':i,'unit':unit,'coordinate':'physical_projection','bounds':{'lower':float(low[i]),'upper':float(high[i])}}
        registry={'schema_version':'2.0-draft','environment_id':'MAbEnvGym','registry_version':'B-mab-physical-research-20261116-1',
                  'evidence_ref':'docs/data_dictionary.csv','observation_size':1970,'action_size':9,'variables':variables}
        validate_registry(registry)
        (ROOT/'configs/environments/mab_physical_registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    with output.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    (ROOT/'configs/environments/dictionary_provenance.json').write_text(json.dumps({'dictionary_sha256':digest,'rows':len(rows),
        'unit_scope':'source declared/inferred only, not field calibrated','mab_reshape':'index=19+13*axial_node+compartment; 150 nodes, 13 compartments; last index1969 separate integral',
        'mab_control_pair':{'observation_index':17,'action_index':7,'equation':'dh/dt=(F_in_buffer-F_out_buffer)/Ac_buffer','source':source(helper,'dhdt =')},
        'unconfirmed':'q1/q2 state units, accumulation integral physical mass conversion, reliable process ranges and real sensor equivalence'},indent=2)+'\n')
    print(f'Wrote {len(rows)} dictionary entries; {digest}')

if __name__=='__main__':main()
