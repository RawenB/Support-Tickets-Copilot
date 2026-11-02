import express from 'express';
import cors from 'cors';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import { tickets } from './data/tickets.js';
import type { CreateTicketInput, Ticket, UpdateTicketStatusInput } from './types.js';

const app = express();
const projectRoot = fileURLToPath(new URL('../../../', import.meta.url));
const predictScriptPath = fileURLToPath(new URL('../../../scripts/predict_priority.py', import.meta.url));

app.use(cors());
app.use(express.json());

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', service: 'tickets-api' });
});

app.post('/api/auth/login', (req, res) => {
  const { email, password } = req.body as { email?: string; password?: string };

  if (email === 'admin@ticketscopilot.com' && password === 'admin123') {
    return res.json({
      user: {
        id: 'admin-1',
        name: 'Admin User',
        email,
        role: 'admin',
      },
      token: 'demo-token-123',
    });
  }

  return res.status(401).json({ message: 'Invalid credentials' });
});

app.get('/api/tickets', (_req, res) => {
  res.json({ data: tickets, total: tickets.length });
});

app.post('/api/ai/predict', (req, res) => {
  const { title, description } = req.body as { title?: string; description?: string };

  if (!title && !description) {
    return res.status(400).json({ message: 'Title or description is required' });
  }

  const child = spawn('python3', [predictScriptPath], {
    cwd: projectRoot,
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
  });

  let stdout = '';
  let stderr = '';

  child.stdin.write(JSON.stringify({ title: title ?? '', description: description ?? '' }));
  child.stdin.end();

  child.stdout.on('data', (chunk) => {
    stdout += chunk.toString();
  });

  child.stderr.on('data', (chunk) => {
    stderr += chunk.toString();
  });

  child.on('close', (code) => {
    if (code !== 0) {
      return res.status(500).json({
        message: 'Prediction failed',
        details: stderr || 'Unknown prediction error',
      });
    }

    try {
      const payload = JSON.parse(stdout);
      return res.json(payload);
    } catch (error) {
      return res.status(500).json({
        message: 'Invalid prediction payload',
        details: stdout || String(error),
      });
    }
  });
});

app.get('/api/tickets/:id', (req, res) => {
  const ticket = tickets.find((item) => item.id === req.params.id);

  if (!ticket) {
    return res.status(404).json({ message: 'Ticket not found' });
  }

  return res.json({ data: ticket });
});

app.post('/api/tickets', (req, res) => {
  const body = req.body as Partial<CreateTicketInput>;

  if (!body.title || !body.description || !body.customerName) {
    return res.status(400).json({ message: 'Title, description, and customerName are required' });
  }

  const now = new Date().toISOString();
  const newTicket: Ticket = {
    id: `TCK-${Math.floor(1000 + Math.random() * 9000)}`,
    title: body.title,
    description: body.description,
    status: 'open',
    priority: body.priority ?? 'medium',
    customerName: body.customerName,
    createdAt: now,
    updatedAt: now,
  };

  tickets.unshift(newTicket);

  return res.status(201).json({ data: newTicket });
});

app.patch('/api/tickets/:id/status', (req, res) => {
  const ticket = tickets.find((item) => item.id === req.params.id);
  const body = req.body as Partial<UpdateTicketStatusInput>;

  if (!ticket) {
    return res.status(404).json({ message: 'Ticket not found' });
  }

  if (!body.status) {
    return res.status(400).json({ message: 'Status is required' });
  }

  ticket.status = body.status;
  ticket.updatedAt = new Date().toISOString();

  return res.json({ data: ticket });
});

app.get('/', (_req, res) => {
  res.json({ message: 'Tickets Copilot API is running' });
});

export default app;
