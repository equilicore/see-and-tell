from cntell.describe.frame import Describe


def test_describe_prepare():
    describe = Describe(model_name="microsoft/git-base")
    describe.prepare(use_dir='~/.cache/cntell')
    
    
def test_describe_run():
    import requests
    import tempfile

    describe = Describe(model_name="microsoft/git-base")
    describe.prepare(use_dir='~/.cache/cntell')
    
    r = requests.get("https://loremflickr.com/320/240")
    with tempfile.NamedTemporaryFile('wb+') as f:
        f.write(r.content)
        f.seek(0)
        # 'Assemble-on-fly' style
        describe.run(images=f.name)
        
        # Explicit style
        out = describe.run(Describe.Images(images=f.name))
        assert out.captions is not None
    

    