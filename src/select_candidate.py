"""Choose candidates using search results alone, before confirmation evaluation."""
import argparse
import json
from pathlib import Path
from weightlab import write_json


def select(run):
    baseline = json.loads((run/'baseline.json').read_text())
    performance = [('original',baseline)]
    for name in ['lora40','taskhalf']:
        value = json.loads((run/(name+'.json')).read_text())
        if value['guard']['token_nll'] <= baseline['guard']['token_nll']*1.03:
            performance.append((name,value))
    best = min(performance,key=lambda x:x[1]['search']['answer_nll'])[0]
    specialist = json.loads((run/'lora40.json').read_text())
    compressed = []
    for name in ['wanda10','taylor08']:
        value = json.loads((run/(name+'.json')).read_text())
        if (value['search']['answer_nll']<=specialist['search']['answer_nll']*1.05 and
            value['guard']['token_nll']<=specialist['guard']['token_nll']*1.03):
            compressed.append(name)
    result = dict(performance_candidate=best, compression_candidates_passing_search_proxy=compressed,
        data='search and guard_search only', selection_rule='minimum answer NLL with prose NLL <= baseline*1.03',
        compression_rule='answer NLL <= specialist*1.05, prose NLL <= specialist*1.03',
        caveat='Search proxy acceptance is not a final task-quality guarantee; no claim sparse zeros accelerate dense inference.')
    write_json(run/'selection.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    select(parser.parse_args().run)
