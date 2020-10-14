const mongoose = require("mongoose")
const axios = require("axios")
const User = require("./../models/userModel")

exports.signup = async(req, res, next) => {
    try {
        console.log(req.body)
        const doc = await User.create(req.body) //no need to enclose req.body in braces as it is already an object
        console.log(doc)
        res.status(200).json({
            status: "success",
            newUser: doc
        })
    } catch (error) {
        next(error)
    }
}

exports.login = async(req, res, next) => {
    try {
        console.log(req.body, 'mkl')
        const doc = await User.find({ enrollNo: req.body.enrollNo, password: req.body.password })
        console.log(doc)
        res.status(200).json({
            status: "success",
            user: doc
        })
        next()
    } catch (error) {
        next(error)
    }
}

exports.data = async(req, res, next) => {
    try {
        const data = req.body;
        console.log(req.body)
        const name = data[0].name
            //console.log(req.body)
            //console.log(data)
        const id = data[0].id
        let len = data.length
        const startTime = data[0].time
        const endTime = data[len - 1].time
        let happy = 0,
            neutral = 0,
            angry = 0,
            sad = 0,
            fearfull = 0,
            disgust = 0,
            surprised = 0
        console.log(happy, surprised)
        data.forEach(element => {
            console.log(element)
            if (element.emotion == "Happy") {
                happy += 1
            } else if (element.emotion == "Angry") {
                angry += 1
            } else if (element.emotion == "Sad") {
                sad += 1
            } else if (element.emotion == "Neutral") {
                neutral += 1
            } else if (element.emotion == "Fearfull") {
                fearfull += 1
            } else if (element.emotion == "Disgust") {
                disgust += 1
            } else {
                surprised += 1
            }
        })
        console.log(happy)
        const total = happy + sad + angry + neutral + fearfull + disgust + surprised
        console.log(total)
        happy = happy / total * 100
        sad = sad / total * 100
        angry = angry / total * 100
        neutral = neutral / total * 100
        disgust = disgust / total * 100
        surprised = surprised / total * 100
        fearfull = fearfull / total * 100
        const newEmotion = {
            startTime: startTime,
            endTime: endTime,
            happy: happy,
            sad: sad,
            angry: angry,
            neutral: neutral,
            disgust: disgust,
            surprised: surprised,
            fearfull: fearfull
        }

        let query = User.find({ name: name });
        let docs = await query;
        //console.log(docs)
        len = docs[0].emotion.length;
        //console.log(docs);
        docs[0].emotion.push(newEmotion);
        //console.log(docs[0].emotion)
        let query2 = User.findByIdAndUpdate(docs[0].id, { $set: { emotion: docs[0].emotion } });
        const docs2 = await query2
            //console.log(docs2)
        res.status(200).json({
            status: "success",
            docs: docs2
        })
        next()
    } catch (error) {
        console.log(error)
    }
}