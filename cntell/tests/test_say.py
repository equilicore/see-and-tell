from cntell.say.caption import Say

 
def test_say_prepare():
    say = Say()
    say.prepare()
    
    
def test_say_run():
    say = Say()
    say.prepare()
    
    # 'Assemble-on-fly' style
    say.run(text="Hello world")
    
    # Explicit style
    out = say.run(Say.Caption(text="Hello world"))
    assert out.audio is not None
    

    