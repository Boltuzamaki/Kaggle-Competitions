import glob

find_str = "['pred_unpadded']))\n    nodes.append(helper.make_node('ReduceMax', ['input'], ['presence'], axes=[1], keepdims=1))\n    nodes.append(helper.make_node('Mul', ['pred_unpadded', 'presence'], ['output']))"

for f in glob.glob(r'c:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\predicted\test_onnx_task*.py'):
    c = open(f).read()
    if 'pred_unpadded' in c:
        c = c.replace(find_str, "['output']))")
        with open(f, 'w') as out:
            out.write(c)
        print("Reverted", f)
