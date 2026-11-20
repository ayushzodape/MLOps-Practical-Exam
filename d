[1mdiff --git a/.dvc/config b/.dvc/config[m
[1mnew file mode 100644[m
[1mindex 0000000..a268d64[m
[1m--- /dev/null[m
[1m+++ b/.dvc/config[m
[36m@@ -0,0 +1,4 @@[m
[32m+[m[32m[core][m
[32m+[m[32m    remote = localstore[m
[32m+[m[32m['remote "localstore"'][m
[32m+[m[32m    url = ../../dvc-storage[m
[1mdiff --git a/.gitignore b/.gitignore[m
[1mnew file mode 100644[m
[1mindex 0000000..a937f39[m
[1m--- /dev/null[m
[1m+++ b/.gitignore[m
[36m@@ -0,0 +1,5 @@[m
[32m+[m[32mvenv/[m
[32m+[m[32m__pycache__/[m
[32m+[m[32m*.pyc[m
[32m+[m[32mmlruns/[m
[32m+[m[32m.ipynb_checkpoints/[m
[1mdiff --git a/data/raw/.gitignore b/data/raw/.gitignore[m
[1mnew file mode 100644[m
[1mindex 0000000..861fdc7[m
[1m--- /dev/null[m
[1m+++ b/data/raw/.gitignore[m
[36m@@ -0,0 +1 @@[m
[32m+[m[32m/dataset.csv[m
[1mdiff --git a/requirements.txt b/requirements.txt[m
[1mnew file mode 100644[m
[1mindex 0000000..96cd9de[m
[1m--- /dev/null[m
[1m+++ b/requirements.txt[m
[36m@@ -0,0 +1,6 @@[m
[32m+[m[32mpandas[m
[32m+[m[32mscikit-learn[m
[32m+[m[32mdvc[m
[32m+[m[32mpyarrow[m
[32m+[m[32mfeast[m
[32m+[m[32mmlflow[m
