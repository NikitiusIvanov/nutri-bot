from main import message_lenght


def test_message_lenght():
    assert message_lenght(None) is None
    assert message_lenght('') == 0
    assert message_lenght('a') == 1
