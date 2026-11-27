"""D process-isolated scorer; deadlines never authorize an environment step."""
import multiprocessing as mp
import time
import math
from .b05_model import B05Model

def _worker(conn,path,digest):
    try:
        import numpy,sklearn
        if numpy.__version__!='1.26.4' or sklearn.__version__!='1.8.0':raise ValueError('unreviewed model dependency versions')
        model=B05Model.load_trusted(path,digest);conn.send({'ready':True})
        while True:
            message=conn.recv()
            if message is None:break
            ident,packet,context=message
            try:result=model.score_packet(packet,context)
            except Exception:result={'status':'unavailable','reason':'model_error','automatic_execution_blocked':True,'execution_authorized':False}
            conn.send((ident,result))
    except (EOFError,BrokenPipeError):pass
    except Exception:
        try:conn.send({'ready':False})
        except Exception:pass
    finally:conn.close()

class InferenceService:
    def __init__(self,path,digest,*,timeout_seconds=.5,startup_seconds=30):
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (timeout_seconds,startup_seconds)):raise ValueError('finite positive budgets required')
        self.budget=timeout_seconds;self.used=set();self.closed=False
        ctx=mp.get_context('spawn');self.conn,child=ctx.Pipe();self.process=ctx.Process(target=_worker,args=(child,str(path),digest),daemon=True)
        self.process.start();child.close()
        try:
            if not self.conn.poll(startup_seconds) or self.conn.recv()!={'ready':True}:raise RuntimeError('model worker startup failed')
        except Exception:
            self.close();raise RuntimeError('model worker startup failed') from None
    def score(self,request_id,packet,context,*,absolute_deadline=None):
        fail=lambda reason:{'status':'unavailable','reason':reason,'automatic_execution_blocked':True,'execution_authorized':False}
        if not isinstance(request_id,str) or not request_id or request_id in self.used:return fail('duplicate_or_invalid_request')
        if not isinstance(packet,dict) or not isinstance(context,dict):return fail('invalid_packet_or_context')
        if absolute_deadline is not None and (type(absolute_deadline) not in (int,float) or not math.isfinite(absolute_deadline)):return fail('invalid_deadline')
        self.used.add(request_id)
        if self.closed:return fail('worker_unavailable')
        deadline=time.monotonic()+self.budget
        if absolute_deadline is not None:deadline=min(deadline,absolute_deadline)
        if deadline<=time.monotonic():return fail('expired_request')
        start=time.monotonic()
        try:
            self.conn.send((request_id,packet,context))
            if not self.conn.poll(max(0,deadline-time.monotonic())):
                self.close();return fail('inference_timeout')
            ident,result=self.conn.recv()
            if ident!=request_id or time.monotonic()>deadline:
                self.close();return fail('late_or_mismatched_result')
            return {**result,'request_id':request_id,'wall_seconds':time.monotonic()-start,'execution_authorized':False}
        except Exception:
            self.close();return fail('worker_or_result_error')
    def close(self):
        if self.closed:return
        self.closed=True
        if self.process.is_alive():self.process.terminate()
        self.process.join(timeout=1);self.conn.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
