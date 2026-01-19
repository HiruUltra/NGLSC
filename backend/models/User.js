// User Schema for MongoDB
const userSchema = {
  _id: "ObjectId",
  name: "String",
  email: "String (unique index)",
  password: "String (hashed with bcryptjs)",
  role: "String (Student or Admin)",
  createdAt: "Date",
  updatedAt: "Date"
};

// Example User Document
const exampleUser = {
  _id: "507f1f77bcf86cd799439011",
  name: "John Doe",
  email: "john@example.com",
  password: "$2a$10$...", // hashed password
  role: "Student",
  createdAt: "2024-01-19T10:00:00Z",
  updatedAt: "2024-01-19T10:00:00Z"
};

module.exports = {
  userSchema,
  exampleUser
};
