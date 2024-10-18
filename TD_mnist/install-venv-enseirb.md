# Install and share a Python virtualenv at ENSEIRB

1. Connect to a machine at ENSEIRB:

    ssh -J <login>@ssh.enseirb.fr <login>@centospedago

2. Create and activate the virtualenv:

    python3 -m venv ML
    source ML/bin/activate
 
    Your prompt should be changed with (<nom>) at the beginning.
    To deactivate the virtualenv, just use the command: deactivate

3. Install necessary packages with pip:

    For example:
    pip install jupyter
    pip install tensorflow
    pip install matplotlib

4. Change rights so that others can access the virtualenv:

    chmod -R go=rX ML
    (this can take a while if many packages have been installed)

    Your home directory should also be accessible:
    chmod go=+x ~

5. Others can now access this virtualenv:

    (source ~<login>/ML/bin/activate)
    source ~gpoette/ML/bin/activate
