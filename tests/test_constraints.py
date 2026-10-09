import math
import unittest
from convxai.evidence import Evidence, Feature, feature_id
from convxai.constraints import template, validate, checked_generate, statement
from convxai.locales import LANGUAGES, TEXT


def sample(scores=(.6, -.3, .1)):
    return Evidence(tuple(Feature(feature_id(name), name, score) for name, score in zip(('age', 'income', 'height'), scores)), 'iris', 'svm', 'shap', 'local', 0, 'class_a')

class ConstraintsTests(unittest.TestCase):
    def test_all_languages(self):
        e = sample()
        for lang in LANGUAGES:
            self.assertTrue(validate(template(e, lang), e, lang).accepted)
            self.assertEqual(set(TEXT['en']), set(TEXT[lang]))

    def test_sign_reversal_even_with_loose_magnitude_tolerance(self):
        e = sample()
        body = template(e).replace('-0.3000000000', '+0.3000000000')
        self.assertEqual(validate(body, e, epsilon=10).reason, 'sign')

    def test_magnitude(self):
        e = sample()
        self.assertEqual(validate(template(e).replace('+0.6000000000', '+0.7000000000'), e).reason, 'magnitude')

    def test_rank_permutation(self):
        e = sample()
        rows = template(e).splitlines()
        body = '\n'.join(f'{i}. {line.split(". ", 1)[1]}' for i, line in enumerate(reversed(rows), 1))
        self.assertEqual(validate(body, e).reason, 'rank')

    def test_ties_can_swap(self):
        e = sample((.5, -.5, .1))
        chosen = list(e.selected());chosen[0], chosen[1] = chosen[1], chosen[0]
        scores = e.normalised()
        body = '\n'.join(statement(i, f, scores[f.id], 'en') for i, f in enumerate(chosen, 1))
        self.assertTrue(validate(body, e).accepted)

    def test_truncated_subset_keeps_full_denominator(self):
        e = sample()
        body = template(e, k=1)
        self.assertIn('+0.6000000000', body)
        self.assertTrue(validate(body, e, k=1).accepted)

    def test_zero(self):
        e = sample((0, 0, 0))
        self.assertTrue(validate(template(e), e).accepted)

    def test_missing_duplicate_unknown_or_changed_label(self):
        e = sample();body = template(e);rows = body.splitlines()
        for candidate in ('\n'.join(rows[:-1]), '\n'.join([rows[0], '2. '+rows[0].split('. ',1)[1], rows[2]]),
                          body.replace(e.features[0].id, 'f'+'0'*16), body.replace('"age"','"unknown"')):
            self.assertFalse(validate(candidate, e).accepted)

    def test_free_prose_and_numeric_formats_rejected(self):
        e = sample()
        for value in ('This means you will recover.', template(e)+'\nThis causes recovery.',
                      template(e).replace('+0.6000000000','NaN'), template(e).replace('+0.6000000000','+0,6000000000')):
            self.assertFalse(validate(value, e).accepted)

    def test_nonfinite_and_duplicate_source_rejected(self):
        for v in (math.nan, math.inf):
            with self.assertRaises(ValueError): sample((v,.2,.1))
        f=Feature(feature_id('x'),'x',1)
        with self.assertRaises(ValueError): Evidence((f,f),'d','m','e','local')

    def test_roundtrip_label_escaping(self):
        f=Feature(feature_id('a'),'a "quote"\\\nline',1)
        e=Evidence((f,),'d','m','e','local')
        self.assertTrue(validate(template(e),e).accepted)

    def test_bounded_retry_fallback(self):
        calls=[]
        def bad(body,lang,attempt): calls.append(attempt);return 'invented'
        body, check, failures, fallback=checked_generate(sample(),generate=bad,retries=1)
        self.assertTrue(check.accepted);self.assertTrue(fallback)
        self.assertEqual(calls,[0,1]);self.assertEqual(len(failures),2)

    def test_provider_error_fallback(self):
        def bad(*args): raise RuntimeError('offline')
        self.assertTrue(checked_generate(sample(),generate=bad)[3])

    def test_valid_provider_response(self):
        result=checked_generate(sample(),generate=lambda body,*args: body)
        self.assertFalse(result[3]);self.assertEqual(result[2],[])
