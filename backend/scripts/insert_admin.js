const bcrypt = require('bcryptjs');
const { MongoClient } = require('mongodb');

(async function(){
  const uri = 'mongodb://localhost:27017';
  const client = new MongoClient(uri);
  try {
    await client.connect();
    const db = client.db('nglsc');
    const users = db.collection('users');
    const pwd = 'Admin@1234';
    const hash = await bcrypt.hash(pwd, 10);
    const now = new Date();
    const admin = { name: 'Admin User', email: 'admin@school.edu', password: hash, role: 'Admin', createdAt: now };

    const existing = await users.findOne({ email: admin.email });
    if (existing) {
      console.log('Admin already exists:', existing.email);
      await client.close();
      process.exit(0);
    }

    const r = await users.insertOne(admin);
    console.log('Inserted admin id', r.insertedId.toString(), 'passwordPlain=', pwd);
    await client.close();
  } catch (e) {
    console.error('ERR', e);
    process.exit(1);
  }
})();
