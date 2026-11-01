import express from 'express';
import cors from 'cors';

import { tickets } from './data/tickets.js';
import type { CreateTicketInput, Ticket, UpdateTicketStatusInput } from './types.js';

const app = express();

app.use(cors());
app.use(express.json());

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', service: 'tickets-api' });
});

app.get('/api/tickets', (_req, res) => {
  res.json({ data: tickets, total: tickets.length });
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
