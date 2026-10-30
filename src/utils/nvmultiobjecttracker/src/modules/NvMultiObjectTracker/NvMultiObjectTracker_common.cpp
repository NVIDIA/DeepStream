/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifdef _WIN32
#include <chrono>
#else
#include <unistd.h>
#endif

#include "modules/NvMultiObjectTracker/NvMultiObjectTracker_common.hpp"
#include "modules/NvMultiObjectTracker/NvTrackedObject.hpp"

/**
 * @brief Launch all worker threads and worker function.
 * @param[in] numThreads Number of worker threads.
 * @param[in] funcPerThread Function to execute in each thread.
 */
template <typename T> void NvTrackerMultiThreadWorker<T>::initialize(
	const uint numThreads, std::function<void(T)> funcPerThread
)
{
	/// Instantiate threads.
	m_persistThreadRunning = true;
	m_submittedJobCnt = 0;
	m_finishedJobCnt = 0;
	m_funcPerThread = funcPerThread;
	for(uint threadCnt = 0; threadCnt < numThreads; threadCnt++)
	{
		m_threadList.push_back(std::thread(&NvTrackerMultiThreadWorker::threadLoopFunc, this));
#ifdef _WIN32
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
#else
        usleep(1*1000);
#endif
	}

	m_isInit = true;
}

/** @brief Initialize an empty worker queue. */
template <typename T> NvTrackerMultiThreadWorker<T>::NvTrackerMultiThreadWorker()
{
	m_isInit = false;
	m_persistThreadRunning = false;
}

/** @brief Stop and join all persistent worker threads. */
template <typename T> NvTrackerMultiThreadWorker<T>::~NvTrackerMultiThreadWorker()
{
	/// Terminate threads.
	if (m_isInit)
	{
		/// Allow assoc threads to exit.
		m_persistThreadRunning = false;

		/// Clear input queue and notify all waiting threads.
		std::unique_lock<std::mutex> inputLock(m_inputLock);

		m_inputQueue = std::queue<T>();

		inputLock.unlock();
		m_cond4In.notify_all();

		/// Join threads.
		for(uint threadCnt = 0; threadCnt < m_threadList.size(); threadCnt++)
		{
			if(m_threadList[threadCnt].joinable())
			{
				m_threadList[threadCnt].join();
			}
		}
		m_threadList.clear();

		/// Clear output.
		m_finishedJobCnt = 0;
	}
}

/**
 * @brief Submit association input data to queue in main thread.
 * @param[in] inData Job data queued for a worker thread.
 */
template <typename T> void NvTrackerMultiThreadWorker<T>::submitInput(T inData)
{
	m_submittedJobCnt++;

	/// Push input into queue.
	std::unique_lock<std::mutex> inputLock(m_inputLock);
	m_inputQueue.push(std::move(inData));
	inputLock.unlock();
	m_cond4In.notify_one();
}

/** @brief Wait all worker threads to finish in main thread. */
template <typename T> void NvTrackerMultiThreadWorker<T>::waitToFinish()
{
	if(m_submittedJobCnt > 0)
	{
		/// Wait for output.
		while(m_persistThreadRunning)
		{
			std::unique_lock<std::mutex> outputLock(m_outputLock);

			/// If output size is not equal to input, wait until all threads generate output.
			if(m_finishedJobCnt < m_submittedJobCnt)
			{
				m_cond4Out.wait(outputLock);
			}

			/// Check for deinit signal or spurious wakeup or output size is not equal to input!!
			if(!m_persistThreadRunning)
			{
				outputLock.unlock();
				break;
			}
			else if(m_finishedJobCnt < m_submittedJobCnt)
			{
				outputLock.unlock();
				continue;
			}
			else
			{
				m_finishedJobCnt = 0;
				outputLock.unlock();
				break;
			}
		}

		m_submittedJobCnt = 0;
	}
	else
	{
		m_finishedJobCnt = 0;
	}
}

/** @brief Process queued jobs until the worker receives a deinitialization signal. */
template <typename T> void NvTrackerMultiThreadWorker<T>::threadLoopFunc()
{
	while(m_persistThreadRunning)
	{
		/// Fetch and remove one queued input atomically.
		std::unique_lock<std::mutex> inputLock(m_inputLock);
		if(m_inputQueue.empty())
		{
			m_cond4In.wait(inputLock);
		}

		/// Check for spurious wakeup!!
		if(!m_persistThreadRunning)
		{
			inputLock.unlock();
			break;
		}
		else if(m_inputQueue.empty())
		{
			inputLock.unlock();
			continue;
		}

		/// Get an input.
		T in = std::move(m_inputQueue.front());
		m_inputQueue.pop();
		inputLock.unlock();

		m_funcPerThread(in);

		/// Lock the completion count so the waiting thread observes it without concurrent worker updates.
		std::unique_lock<std::mutex> outputLock(m_outputLock);
		m_finishedJobCnt++;
		outputLock.unlock();

		/// Send signal to NvTrackerMultiThreadWorker::waitToFinish().
		m_cond4Out.notify_all();
	}

	/// When deinit, should also send signal to NvTrackerMultiThreadWorker::waitToFinish().
	m_cond4Out.notify_all();
}

/// Explicitly instantiate the worker templates so their methods are linked.
template class NvTrackerMultiThreadWorker<assocDataIn_t>;
template class NvTrackerMultiThreadWorker<DisMahalanobisIn>;
