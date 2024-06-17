tmux \
    new-session  'bash launch_trt_server.sh' \; \
    split-window   'bash launch_app.sh' 
