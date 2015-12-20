__author__ = 'sz372'

import templates


def test_window():
    t = templates.Template('foobar', 0)
    assert 'foobar' == t._window('foobar', 0, e=1)
    t = templates.Template('foobar', 1)
    assert 'foobar' == t._window('foobar', 1, e=1)
    assert 'oobar' == t._window('foobar', 2, e=1)
    t = templates.Template('foobar', 2)
    assert 'fooba' == t._window('foobar', 1, e=1)
    assert 'foobar' == t._window('foobar', 2, e=1)
    t = templates.Template('foobar', 3)
    assert 'foobar' == t._window('foobar', 3, e=1)
    t = templates.Template('foobar', 4)
    assert 'foobar' == t._window('foobar', 4, e=1)
    t = templates.Template('foobar', 5)
    assert 'foobar' == t._window('foobar', 5, e=1)


def test_matches():
    ret = templates.match('foobar', 'foobar')
    assert ret == set([templates.Match('foobar')])

    ret = templates.match('foobar', 'fobar')
    assert ret == set([templates.Match('fobar')])

    ret = templates.match('foobar', 'afo|bar$')
    assert ret == set([templates.Match('fo|bar$'), templates.Match('afo|bar$')])
