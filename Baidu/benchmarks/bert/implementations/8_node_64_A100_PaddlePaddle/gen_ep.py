import os

env_name = 'PADDLE_TRAINER_ENDPOINTS'
env_value = os.getenv(env_name)

node = int(os.getenv('PADDLE_TRAINERS_NUM', 1)) 

if env_value and node > 1:
    ips = [s.strip().split(':')[0] for s in env_value.split(',') if s.strip()] 
    unique_ips = []
    for ip in ips:
        if ip not in unique_ips:
            unique_ips.append(ip)
 
    if len(unique_ips) > 1:
        ports = os.getenv('PADDLE_TRAINER_PORTS') 
        assert ports  
        ports = [p.strip() for p in ports.split(',') if p.strip()] 
        new_ep = []  
        for ip in unique_ips:
            for port in ports:
                new_ep.append('{}:{}'.format(ip, port))
        print(','.join(new_ep))
    else:
        print(env_value)
