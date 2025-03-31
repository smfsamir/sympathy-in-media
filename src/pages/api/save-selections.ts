import fs from "fs";
import path from "path";
import { NextApiRequest, NextApiResponse } from "next";

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === "POST") {
    const email  = req.body.email.split("@");
    const user = email[0];
    const article = req.body.article;
    const fileName = user + "_" + article + "_selections.json";
    const dirPath = path.join(process.cwd(), "data", "annotations");
    const filePath = path.join(dirPath, fileName); // TODO: rewrite this so it saves the username as well as the person/article being annotated
    // const filePath = path.join(process.cwd(), "", "selections.json");

    try {
      // Save data to a JSON file
      fs.writeFileSync(filePath, JSON.stringify(req.body, null, 2), "utf8");
      return res.status(200).json({ message: "Selections saved successfully" });
    } catch (error) {
      return res.status(500).json({ message: "Error saving selections", error });
    }
  }

  return res.status(405).json({ message: "Method not allowed" });
}
