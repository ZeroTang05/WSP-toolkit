"""Container-only execution port of OfficeBench actions and official checks.

Source semantics: utils/env.py at b978b808667c32b52ce19a67ce1def1de9ae02b7.
Changes: JSON instead of eval, subprocess argv boundaries, fresh containers,
explicit timeouts. Upstream app implementations and scoring functions are loaded
unchanged. Network calls are unavailable in the actor container.
"""
import contextlib
import io
import json
from pathlib import Path
import posixpath
import subprocess
import sys


def main(request):
    if request['mode']=='score':
        import utils.evaluate as evaluation
        checks=[]
        allowed={'evaluate_contain','evaluate_not_contain','evaluate_file_exist',
            'evaluate_file_not_exist','evaluate_diff_contain_text','evaluate_excel_cell_value',
            'evaluate_excel_cell_comparator','evaluate_exact_match','evaluate_calendar_no_overlap'}
        for item in request['checks']:
            if item['function'] not in allowed:raise ValueError('Unknown official check')
            try:
                ok=bool(getattr(evaluation,item['function'])(request['testbed'],item['args']))
                checks.append(dict(function=item['function'],passed=ok,error=None))
            except Exception as error:
                checks.append(dict(function=item['function'],passed=False,error=type(error).__name__+': '+str(error)))
        return dict(all_pass=all(c['passed'] for c in checks),checks=checks)
    import apps
    if request['mode']=='describe':
        return dict(apps={name:dict(intro=module.INTRO,actions={action:getattr(impl,'DEMO','')
            for action,impl in apps.AVAILABLE_ACTIONS[name].items()}) for name,module in apps.AVAILABLE_APPS.items()},
            system_actions=['switch_app','finish_task','got_stuck'])
    action={k:(v[0] if isinstance(v,list) and len(v)==1 else v) for k,v in request['action'].items()}
    app=action['app'];operation=action['action'];state=dict(request['state']);done=False;command=None
    if operation=='switch_app':app='system'
    if app=='system':
        if operation=='switch_app':
            if action['target_app'] not in apps.AVAILABLE_APPS:raise ValueError('Unknown application')
            state['current_app']=action['target_app'];observation='Successfully switched to app: '+action['target_app']
        elif operation=='finish_task':
            # Preserve upstream answer-text quote stripping, without shell echo.
            answer=str(action.get('answer','None')).replace('"','').replace("'",'')
            Path('/testbed/data').mkdir(exist_ok=True);Path('/testbed/data/answer.txt').write_text(answer+'\n')
            done=True;observation='Task finished'
        elif operation=='got_stuck':done=True;observation='Task failed'
        else:raise ValueError('Unknown system action')
    elif app=='shell':command=action['command']
    elif app=='llm':
        return dict(state=state,done=False,success=False,observation='LLM application disabled in the declared network-free 4B-only execution protocol.')
    else:command=apps.AVAILABLE_ACTIONS[app][operation].construct_action(state['workdir'],args=action)
    if command is not None:
        if isinstance(command,list):command=' '.join(command)
        is_cd=app=='shell' and command.startswith('cd')
        new_path=None
        if is_cd:
            new_path=posixpath.normpath(posixpath.join(state['workdir'],command[command.index('cd ')+3:].strip()))
            command='cd '+new_path
        result=subprocess.run(['bash','-c',command],cwd=state['workdir'],capture_output=True,text=True,timeout=60)
        success=result.returncode==0;observation=result.stdout.split('OBSERVATION:')[-1].strip()
        if is_cd and success:state['workdir']=new_path
        if not observation:observation=(result.stderr or 'Successfully executed command').strip()
        return dict(state=state,done=False,success=success,observation=observation[:20000],output_truncated=len(observation)>20000)
    return dict(state=state,done=done,success=True,observation=observation)


if __name__=='__main__':
    captured=io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):result=main(json.load(sys.stdin))
    except Exception as error:result=dict(success=False,done=False,error=type(error).__name__+': '+str(error),observation='Malformed or failed action')
    print('IDEA4_RESULT:'+json.dumps(result))
