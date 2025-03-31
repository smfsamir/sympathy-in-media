import fs from "fs";
import path from "path";
import { NextApiRequest, NextApiResponse } from "next";

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === "POST") {
    const email  = req.body.email.split("@");
    const user = email[0];
    const article = req.body.article;
    const fileName = user + "_" + article;
    const dirPath = path.join(process.cwd(), "data", "annotations");
    const filePath = path.join(dirPath, fileName); 

    try {
      // Save data to a JSON file
      fs.writeFileSync(filePath, JSON.stringify(req.body, null, 2), "utf8"); // can change this to req.body.selections if we dont want user/article objects
      return res.status(200).json({ message: "Selections saved successfully" });
    } catch (error) {
      return res.status(500).json({ message: "Error saving selections", error });
    }
  }

  return res.status(405).json({ message: "Method not allowed" });
}
