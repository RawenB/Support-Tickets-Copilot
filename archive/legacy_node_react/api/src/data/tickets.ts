import type { Ticket } from '../types.js';

export const tickets: Ticket[] = [
  {
    id: 'TCK-1001',
    title: 'Login issue on mobile app',
    description: 'Customers report being unable to log in after password reset.',
    status: 'open',
    priority: 'high',
    customerName: 'Amina Hassan',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
  {
    id: 'TCK-1002',
    title: 'Billing discrepancy',
    description: 'Customer was charged twice for the same invoice.',
    status: 'in_progress',
    priority: 'urgent',
    customerName: 'Daniel Kim',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
];
