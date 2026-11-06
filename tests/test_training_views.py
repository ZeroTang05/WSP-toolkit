import json
import tempfile
import sys
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prepare_training_views import read_jsonl


class TrainingViewTests(unittest.TestCase):
    def test_unicode_paragraphs_do_not_split_jsonl_records(self):
        rows=[dict(prompt='line\u2028paragraph',answer='value\u0085text'),dict(prompt='second',answer='2')]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'rows.jsonl'
            path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
            self.assertEqual(read_jsonl(path),rows)


if __name__=='__main__':unittest.main()
