#!/bin/bash

container_name=$1

if [ -z "$CONTAINER_ID" ]; then
    echo "Error: No container ID or name provided."
    exit 1
fi

container_ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$container_name")
container_iface=$(docker exec "$container_name" ip -o link show | awk -F': ' '{print $2}' | grep 'eth')

echo $container_ip
echo $container_iface