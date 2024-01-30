# Unit testing
- Run all unit testing
```bash
cd tests && pytest -s 
```
- Run unit testing under a single folder
```bash
cd tests && pytest -s {dir}
```
- Rerun the test cases that failed in the last run
```bash
cd tests && pytest -s --lf
```