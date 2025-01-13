module.exports = {
  apps: [{
    name: 'paper-table-backend',
    script: 'src/server.py',
    interpreter: 'python3',
    interpreter_args: '-u',
    args: '--no-backend',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      PORT: 4305,
      PYTHONUNBUFFERED: 1,
      PYTHONIOENCODING: 'utf-8'
    }
  }]
}