__author__ = 'sz372'

import textlines

def test_filter():
    l = textlines.Line([(12,0,5,5),(0,0,5,5),(6,0,5,5)], ['p','t','m'])
    l.filter()
    print l.components
    assert l.components == [(0,0,5,5),(6,0,5,5),(12,0,5,5)]
    assert l.text == ['t','m','p']