Docs
==========


## Building the docs

To build and preview the docs run the following commands:

```buildoutcfg
cd docs
pip3 install -r requirements.txt
make html
python3 -m http.server 8082 --bind ::
```

Now you should be able to view the docs in your browser at the link provided in your terminal.

To reload the preview after making changes, rerun:

```
make html
python3 -m http.server 8082 --bind ::
```
