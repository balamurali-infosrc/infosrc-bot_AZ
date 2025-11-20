# #!/bin/bash
# service ssh start

# # Start the aiohttp bot
# gunicorn main:APP \
#     --worker-class aiohttp.GunicornWebWorker \
#     --bind 0.0.0.0:$PORT \
#     --workers 1 \
#     --timeout 600

# #!/bin/bash
# echo "Starting Azure App Service Bot (Gunicorn + aiohttp)"

# gunicorn main:APP \
#   --worker-class aiohttp.worker.GunicornWebWorker \
#   --bind=127.0.0.1:$PORT \
#   --timeout 600


#!/bin/bash
echo "Starting (Gunicorn + aiohttp)"
gunicorn main:APP --worker-class aiohttp.worker.GunicornWebWorker --bind 0.0.0.0:${PORT}

#!/bin/bash

# echo "Starting Azure App Service Bot (Gunicorn + aiohttp)"

# gunicorn main:APP \
#     --worker-class aiohttp.worker.GunicornWebWorker \
#     --bind=127.0.0.1:${PORT:-8000} \
#     --timeout 600


