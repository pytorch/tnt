# displays parameterized arguments in pytest output
def pytest_make_parametrize_id(config, val, argname):
    return f"{argname}={val}"
