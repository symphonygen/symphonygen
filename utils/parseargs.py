"""
Provides easy CLI integration using the first letters of arguments.
Adapted from pip package `parseargs` to include bool support (store_true), and removed -h for help.
"""
import argparse
import inspect

def parseargs(f):
    argspec = inspect.getfullargspec(f)

    argnames = argspec[0]
    kwdefaults = list() if argspec[3] is None else argspec[3]
    poslen = len(argnames) - len(kwdefaults)
    posnames = argnames[:poslen]
    annotations = argspec[6]

    parser = argparse.ArgumentParser(description=('signature = ' + str(inspect.signature(f))), add_help=False)

    # Add args, using defaults if provided
    for i in range(len(argnames)):
        name = argnames[i]
        typ = annotations.get(name, str)
        if i - poslen >= 0:
            default = kwdefaults[i - poslen]
            if typ is bool:
                if default is True:
                    print(f"Warning: Boolean flag {name}'s default value {default} is ignored")
                parser.add_argument(f'-{name[0]}', f'--{name}', action='store_true')
            else:
                parser.add_argument(f'-{name[0]}', f'--{name}', required=False, type=typ, default=default)
        else:
            parser.add_argument(name, type=typ)

    # Compile parser
    args = parser.parse_args()
    kwargs = vars(args).copy()
    posargs = [kwargs[name] for name in posnames]
    for name in posnames:
        del kwargs[name]

    f(*posargs, **kwargs)
