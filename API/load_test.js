import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { BASE64 } from './base64-img.js'


export let options = {
    stages: [
        { duration: '1m', target: 1 }, // simulate ramp-up of traffic from 1 to 100 users over 5 minutes.
        { duration: '1m', target: 1 }, // stay at 100 users for 10 minutes
        { duration: '1m', target: 1 }, // ramp-down to 0 users
    ],
    thresholds: {
        http_req_duration: ['p(99)<1500'], // 99% of requests must complete below 1.5s
        'logged in successfully': ['p(99)<1500'], // 99% of requests must complete below 1.5s
    },
};

const URL = 'http://201.216.239.79:5000/test-liveness';

export default () => {

    let authHeaders = {
        headers: {
            'x-api-key': `1a0bb17e-cfe9-4283-b59a-cd97db897231`,
            'Content-Type': 'application/json',
        },
    };

    console.log(JSON.stringify(http.post(`${URL}`, JSON.stringify({ "img": BASE64 }), authHeaders).json()));

    sleep(1);
};