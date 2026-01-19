module.exports = (db) => {
  const express = require('express');
  const router = express.Router();
  const { verifyToken, requireRole } = require('../middleware/auth');

  /**
   * @route   GET /api/users/me
   * @desc    Get current logged-in user
   * @access  Private
   */
  router.get('/me', verifyToken, (req, res) => {
    res.status(200).json({
      message: 'User data retrieved',
      user: req.user
    });
  });

  /**
   * @route   GET /api/users
   * @desc    Get all users (Admin only)
   * @access  Private/Admin
   */
  router.get('/', verifyToken, requireRole('Admin'), async (req, res) => {
    try {
      const users = await db.collection('users').find({}).toArray();
      res.status(200).json({
        message: 'Users retrieved successfully',
        count: users.length,
        users: users.map(u => ({
          id: u._id,
          name: u.name,
          email: u.email,
          role: u.role,
          createdAt: u.createdAt
        }))
      });
    } catch (error) {
      res.status(500).json({ error: 'Failed to retrieve users' });
    }
  });

  /**
   * @route   PUT /api/users/:id
   * @desc    Update user information
   * @access  Private
   */
  router.put('/:id', verifyToken, (req, res) => {
    res.status(200).json({
      message: 'User updated successfully',
      userId: req.params.id
    });
  });

  /**
   * @route   DELETE /api/users/:id
   * @desc    Delete user (Admin only)
   * @access  Private/Admin
   */
  router.delete('/:id', verifyToken, requireRole('Admin'), (req, res) => {
    res.status(200).json({
      message: 'User deleted successfully',
      userId: req.params.id
    });
  });

  return router;
};
