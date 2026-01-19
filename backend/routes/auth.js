module.exports = (db) => {
  const express = require('express');
  const router = express.Router();
  const authController = require('../controllers/authController');

  /**
   * @route   POST /api/auth/register
   * @desc    Register a new user
   * @access  Public
   */
  router.post('/register', (req, res) => {
    authController.register(req, res, db);
  });

  /**
   * @route   POST /api/auth/login
   * @desc    Login user and get JWT token
   * @access  Public
   */
  router.post('/login', (req, res) => {
    authController.login(req, res, db);
  });

  /**
   * @route   GET /api/auth/verify
   * @desc    Verify JWT token
   * @access  Public
   */
  router.get('/verify', (req, res) => {
    authController.verifyEmail(req, res);
  });

  return router;
};
