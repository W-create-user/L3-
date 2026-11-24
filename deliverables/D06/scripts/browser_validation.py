"""Browser interaction/video acceptance for the separately running D console."""
import argparse,json
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:5056')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    events=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        context=browser.new_context(viewport={'width':1120,'height':950},record_video_dir=str(a.output/'video'),record_video_size={'width':1120,'height':950})
        page=context.new_page();page.goto(a.url);page.wait_for_selector('#mode')
        def state():return page.request.get(a.url+'/status').json()
        def action(text,expected_status=200):
            with page.expect_navigation() as nav:page.get_by_role('button',name=text,exact=True).click()
            response=nav.value
            assert response.status==expected_status,(text,response.status)
            if expected_status!=200:page.goto(a.url)
            current=state();events.append({'action':text,'status':expected_status,'state':current})
            page.wait_for_timeout(300)
            return current
        assert state()['step_calls']==0
        page.screenshot(path=str(a.output/'initial.png'))
        action('自主执行一轮',409);assert state()['step_calls']==0
        for _ in range(6):action('受限采样一步')
        assert state()['step_calls']==6
        # Retain exactly the form used by the first confirmation for replay.
        old=page.locator('form').evaluate('(f)=>Object.fromEntries(new FormData(f))')
        action('确认有效历史');assert state()['step_calls']==6
        duplicate=page.request.post(a.url+'/confirm',form=old)
        assert duplicate.status==409
        events.append({'action':'replayed confirmation form','status':duplicate.status,'state':state()})
        action('自主执行一轮');assert state()['step_calls']==7
        action('测试越界候选拦截');assert state()['step_calls']==7;assert state()['mode']=='takeover_requested'
        page.screenshot(path=str(a.output/'takeover.png'))
        action('自主执行一轮',409);action('确认有效历史',409)
        old_episode=state()['episode_id'];action('明确开始新 episode')
        assert state()['episode_id']!=old_episode and state()['step_calls']==7
        for _ in range(6):action('受限采样一步')
        action('确认有效历史');assert state()['step_calls']==13
        action('自主执行一轮');assert state()['step_calls']==14
        page.screenshot(path=str(a.output/'restored.png'))
        # Exercise actual terminal state, retaining every completed step.
        for _ in range(65):
            if state()['mode']=='takeover_requested':break
            action('自主执行一轮')
        assert state()['mode']=='takeover_requested'
        ended_count=state()['step_calls'];action('确认有效历史',409)
        assert state()['step_calls']==ended_count
        page.screenshot(path=str(a.output/'terminal.png'))
        video=page.video;context.close();browser.close()
        video_path=video.path()
    report={'passed':True,'backend':'real SMPL Atropine','browser':'Chromium','simulated_operator':True,
            'human_factor_study':False,'events':events,'video':str(video_path),
            'terminal_step_calls':ended_count,'screenshots':'internal layout QA only; not user deliverables'}
    (a.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':True,'browser_actions':len(events),'terminal_step_calls':ended_count,'video':str(video_path)}))

if __name__=='__main__':main()
