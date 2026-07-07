import re
with open(r'repairs\user_code\task377.py', 'r') as f:
    text = f.read()

# I will just write a loop to do this without regex to avoid nested parenthesis issues
def fix_reshape(line):
    if "nn('Reshape', [" in line and "shape=" in line:
        # e.g. nn('Reshape', ['I_idx_1'], ['I_idx_raw'], shape=addK('sh3030', [30, 30], I64))
        # -> nn('Reshape', ['I_idx_1', addK('sh3030', [30, 30], I64)], ['I_idx_raw'])
        prefix, rest = line.split("shape=")
        # prefix: "nn('Reshape', ['I_idx_1'], ['I_idx_raw'], "
        # rest: "addK('sh3030', [30, 30], I64))\n"
        shape_arg = rest.strip()
        if shape_arg.endswith(")"):
            shape_arg = shape_arg[:-1]
        
        # parse prefix to find input list and output list
        p1 = prefix.index("[")
        p2 = prefix.index("]", p1)
        ins = prefix[p1+1:p2] # "'I_idx_1'"
        
        p3 = prefix.index("[", p2)
        p4 = prefix.index("]", p3)
        outs = prefix[p3+1:p4] # "'I_idx_raw'"
        
        new_ins = ins + ", " + shape_arg
        
        return line[:p1] + "[" + new_ins + "], [" + outs + "])\n"
    return line

with open(r'repairs\user_code\task377.py', 'r') as f:
    lines = f.readlines()

new_lines = [fix_reshape(l) for l in lines]

with open(r'repairs\user_code\task377.py', 'w') as f:
    f.writelines(new_lines)
