"""D-owned loopback web console for explicit simulation supervision."""
from secrets import token_urlsafe,compare_digest
from threading import Lock
from flask import Flask,request,abort,redirect,render_template_string,jsonify

PAGE='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>L3 D 监管实验台</title><style>body{font:17px sans-serif;background:#f2f5f8;color:#15283d;margin:0}main{max-width:950px;margin:32px auto;padding:25px;background:white;border-radius:12px}h1{font-size:28px}section{padding:16px;background:#f5f8fc;margin:16px 0;border-radius:8px}.state{font-size:23px;color:#155e75}label{display:block;margin:9px 0}input{padding:8px;width:85%;font:inherit}button{padding:10px 15px;margin:6px 4px 4px 0;font:inherit;background:#155e75;color:white;border:0;border-radius:5px}button.stop{background:#a33131}pre{white-space:pre-wrap;overflow-wrap:anywhere}small{color:#526070}</style>
<main><h1>L3 D 监管实验台</h1><p>本地 SMPL 仿真 · 孤立森林与参考残差 · 人工确认恢复</p>
<section><div class="state" id="mode">{{state.mode}}</div><p>实际 step 调用：<b id="steps">{{state.step_calls}}</b>　有效采样：{{state.samples}} / 6</p><small>模型概率仅指人工传感器偏置；不代表物理安全概率。初始受限采样每次都需人工提交，并经过硬规则审核。</small></section>
<form method="post"><input type="hidden" name="token" value="{{token}}"><input type="hidden" name="revision" value="{{revision}}">
<label>操作人<input name="operator" value="模拟实验操作人" required></label><label>操作原因<input name="reason" value="检查当前仿真状态与模型记录" required></label>
<button formaction="/sample">受限采样一步</button><button formaction="/confirm">确认有效历史</button><button formaction="/cycle">自主执行一轮</button>
<button class="stop" formaction="/probe">测试越界候选拦截</button><button formaction="/restart">明确开始新 episode</button></form>
<section><h2>最新风险证据</h2><pre id="risk">{{risk}}</pre></section>
<section><h2>最近状态记录</h2><pre>{{events}}</pre></section><p><small>严重风险后停止推进。冻结的同一条观测不能累计成三条恢复样本；新 episode 与原 episode 分别记录。所有操作均为模拟实验，不是志愿者或工厂安全认证。</small></p></main></html>'''

def create_app(session):
    app=Flask(__name__);token=token_urlsafe(32);revision=0;lock=Lock()
    @app.get('/')
    def dashboard():
        import json
        with lock:
            return render_template_string(PAGE,state=session.status(),token=token,revision=revision,
                risk=json.dumps(session.status()['risk'],ensure_ascii=False,indent=2),
                events=json.dumps(session.events[-5:],ensure_ascii=False,indent=2))
    @app.get('/status')
    def status():
        with lock:return jsonify(session.status())
    @app.post('/<operation>')
    def operate(operation):
        nonlocal revision
        if operation not in ('sample','confirm','cycle','probe','restart'):abort(404)
        if not compare_digest(request.form.get('token','').encode(),token.encode()):abort(403)
        with lock:
            if request.form.get('revision')!=str(revision):abort(409,'表单已使用或已过期')
            revision+=1
            try:
                operator=request.form.get('operator');reason=request.form.get('reason')
                session._operator(operator,reason)
                if operation in ('sample','confirm','restart'):getattr(session,operation)(operator=operator,reason=reason)
                elif operation=='cycle':session.cycle()
                else:session.reject_probe()
            except ValueError:abort(400,'需要有效操作人、原因和恢复历史')
            except RuntimeError:abort(409,'当前风险或会话状态不允许此操作')
        return redirect('/',code=303)
    @app.after_request
    def no_cache(response):
        response.headers['Cache-Control']='no-store';return response
    return app

if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    from l3_system.execution import EnvironmentAdapter
    from .residual_detector import ResidualDetector
    from .supervised_session import SupervisedSession
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--expected-model-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=5056);a=p.parse_args()
    model=ResidualDetector.load_trusted(a.model,a.expected_model_sha256)
    with EnvironmentAdapter(max_steps=60) as backend:
        session=SupervisedSession(backend,model,a.output);session.start()
        create_app(session).run(host='127.0.0.1',port=a.port,threaded=True)
