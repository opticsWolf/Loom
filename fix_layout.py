import io, re
p = 'rust/navette-py/src/synthesis_pipeline.rs'
s = io.open(p, encoding='utf-8').read()

def take(start_marker, end_marker):
    global s
    i = s.find(start_marker)
    assert i != -1, start_marker
    k = s.find(end_marker, i) + len(end_marker)
    block = s[i:k]
    s = s[:i] + s[k:]
    return block

b1 = take('/// One film for `run_design`', '}\n')
# b1 ends at struct close; run_design pyfunction follows separately
b2 = take('/// End-to-end design run over evaluated arrays', '    result_to_dict(py, &res, PyDesignStack::from_inner(stack))\n}\n')
b3 = take('/// Shared run-result assembly', '    Ok(d.unbind())\n}\n')
cluster = b1 + '\n' + b2 + '\n' + b3
anchor = 'impl PyDesignStack {'
assert anchor in s
s = s.replace(anchor, cluster + '\n' + anchor, 1)
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('CLUSTERED')
