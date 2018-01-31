from lz import *

temp = open('conf_meta.py', 'r').read()
temp = string.Template(temp)
procs = []
for chs in range(30):
    conf_str = temp.substitute(CHS=str(chs))

    with open('conf.py', 'w') as f:
        f.write(conf_str)
    msg = proc = shell('python cifar.py', block=False)
    procs.append(proc)
    time.sleep(10)

for proc in procs:
    msg = proc.communicate()
    msg = [msg_.decode('utf-8') for msg_ in msg]
    print('stdout', msg[0])
    print('stderr', msg[1])

