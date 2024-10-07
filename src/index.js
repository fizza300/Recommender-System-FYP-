const express = require("express");
const path = require("path");
const app = express();
const hbs = require("hbs");
const LogInCollection = require("./mongodb");

// Paths
const templatePath = path.join(__dirname, '../templates');
const publicPath = path.join(__dirname, '../public');

// Middleware
app.use(express.json());
app.set('view engine', 'hbs');
app.set('views', templatePath);
app.use(express.static(publicPath));
app.use(express.urlencoded({ extended: false }));



// Routes
app.get('/', (req, res) => {
    res.render('login');
});
app.get('/signup', (req, res) => {
    res.render('signup');
});

app.get('/home', (req, res) => {
    res.render('home');
});
app.get('/songs', (req, res) => {
    res.render('songs');
});
app.get('/movies', (req, res) => {
    res.render('movies');
});

app.get('/emotions', (req, res) => {
    res.render('emotions');
});

app.get('/booking', (req, res) => {
    res.render('booking');
});
app.get('/novels', (req, res) => {
    res.render('novels');
});
app.get('/login', (req, res) => {
    res.render('login');
});
app.get('/prediction', (req, res) => {
    res.render('prediction');
});

// Signup handler
app.post('/signup', async (req, res) => {
    try {
        const { name, email, password, gender } = req.body;
        const data = { name, email, password, gender };
        await LogInCollection.insertMany([data]);
        console.log("User signed up: ", data);
        res.render("pf");
    } catch (e) {
        console.error("Error during signup: ", e.message);
        res.status(500).send("Error during signup: " + e.message);
    }
});

// Login handler
app.post('/login', async (req, res) => {
    try {
        const check = await LogInCollection.findOne({ name: req.body.name });

        if (check.password === req.body.password) {
            res.render("home");
        } else {
            res.send("incorrect password");
        }
    } catch (e) {
        res.send("wrong details");
    }
});


// Start the server
app.listen(3000, () => {
    console.log("Server running on http://localhost:3000");
});
