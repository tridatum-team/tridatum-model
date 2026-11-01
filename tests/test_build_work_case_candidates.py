import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_work_case_candidates import article, stem  # noqa: E402

RULE = ('지침\n제1조(목적) 목적.\n제2조(정의) 정의 ① 가.\n② 나.\n제4장 이행\n제3조(이행) 이행.\n'
        '부칙 <제1호>\n제1조(시행일) 시행.\n<별표>\n표 1행\n※ 비고')


class ArticleCutTest(unittest.TestCase):
    def test_article_stops_at_next_article_chapter_and_addenda(self):
        self.assertEqual(article(RULE, '제1조(목적)'), '제1조(목적) 목적.')
        self.assertEqual(article(RULE, '제2조(정의)'), '제2조(정의) 정의 ① 가.\n② 나.')
        self.assertEqual(article(RULE, '제3조(이행)'), '제3조(이행) 이행.')

    def test_annex_runs_to_end_and_missing_heading_raises(self):
        self.assertEqual(article(RULE, '<별표>'), '<별표>\n표 1행\n※ 비고')
        with self.assertRaises(KeyError):
            article(RULE, '제9조(없음)')

    def test_family_stem_joins_decree_and_special_rules_to_their_act(self):
        self.assertEqual(stem('채용절차의 공정화에 관한 법률 시행령'), '채용절차의 공정화에 관한 법률')
        self.assertEqual(stem('특정조달을 위한 국가를 당사자로 하는 계약에 관한 법률 시행령 특례규정'),
                         '국가를 당사자로 하는 계약에 관한 법률')
        self.assertEqual(stem('예산성과금 규정'), '예산성과금 규정')


if __name__ == '__main__':
    unittest.main()
