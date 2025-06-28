#!/bin/bash

# Script taken with modifications from https://dev.to/bgord/multiple-postgres-databases-in-a-single-docker-container-417l

set -e
set -u

function create_user_and_database() {
	local database=$1
	echo "  Creating user and database '$database'"
	psql -d postgres -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
	    CREATE DATABASE "$database";
	    GRANT ALL PRIVILEGES ON DATABASE "$database" TO "$POSTGRES_USER";
EOSQL
}

for var in $(env | grep '^POSTGRES_CREATE_DB_' | awk -F '=' '{print $2}'); do
	echo "Creating database: $var"
	create_user_and_database $var
done